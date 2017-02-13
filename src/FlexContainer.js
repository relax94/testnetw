import React, {Component} from 'react'


export class FlexContainer extends Component {

    render() {
        return (<div className="flex-container">
            {this.props.children}
        </div>)
    }
}